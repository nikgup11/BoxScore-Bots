// lib/api.ts
import { supabase } from './supabase'

export async function getProjections() {
  const { data, error } = await supabase.from('projections').select('*')
  if (error) throw error
  return data
}